import click
import regression as reg

@click.group()
def cli():
    pass

@click.command()
@click.argument("name")
def regression(name):
    method = getattr(reg, name)
    method.execute()

cli.add_command(regression)

if __name__ == "__main__":
    cli()

