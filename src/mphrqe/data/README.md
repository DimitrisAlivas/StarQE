# Instructions for Preprocessing #

If desired, it is possible to run the preprocessing, which generates the query sampls from the original data graphs. This process will take a few hours.

Alternatively, you can download the preprocessed data. Instructions are on [the main readme](../../../README.md)

## Requirements ##

* Ubuntu 18.04 (later appears to work as well) or Mac OSX
    * Windows is not tested
* Docker (for the use of anzograph) installed. Make sure you also follow post installation steps for your platform. (e.g., that you are in the docker user group)
    * See also the anzograph-docker documentation https://docs.cambridgesemantics.com/anzograph/v2.3/userdoc/install-docker.htm
* Python requirements are the same as for the [model code](../../../README.md).

To perform this preprocessing, you need to have at least 30GB of memory available.



### Install anzograph ###

Install anzograph following the instructions https://docs.cambridgesemantics.com/anzograph/v2.3/userdoc/deploy-docker.htm

While installing anzograph, note the following:

1. You need to register for anzograph https://info.cambridgesemantics.com/anzograph/download . After registration go to https://docs.cambridgesemantics.com/anzograph/v2.3/userdoc/register-license.htm to find instructions on how to upgrade your license. You need to have at least the version with a 16GB RAM limit. Make sure to restart your docker container after applying the license key.
2. When starting the docker container, we have to specify an additional open port and a shared directory. This shared directory is a directory within the project, make sure to replace {absolute_project_root} with the absolute path to the root of the project.

```bash
docker run -d -p 8080:8080 -p 8256:8256 -v {absolute_project_root}/data/triple_data/:/opt/shared-files --name=anzograph cambridgesemantics/anzograph:latest
```

By default these are the credentials for anzograph

>    Username: admin  
>    Password: Passw0rd1


In principle all of the below should also work with other RDF*/SPARQL* aware triple stores. Unfortunately, we found several triple stores that either did not handle them in the standard way or went out of memory with the queries we use.

## Downloading the triple data splits ###
Download the data. This will create three files in `/data/triple_data`, each containing one of the splits.

```python executables/main.py preprocess download_wd50k```

## Put the data into the triple store ##

To automatically upload the data to anzograph, execute

```python executables/main.py preprocess initialize```

## Create the queries ##

Creating the queries goes in two steps. First we create them in textual form:

```python executables/main.py preprocess sparql```

Then we convert this textual form to a more compact binary format

```python executables/main.py preprocess convert```


## Continue training models ##

Now all the data is created, and the directory `/data/binaryQueries` should have all generated queries in it.
You can now start the training, validation, and testing with these queries.
